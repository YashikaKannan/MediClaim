from __future__ import annotations

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
PROCESSED_DIR = ROOT / "processed"
MASTER_PATH = ROOT / "provider_master_table.csv"
STREAMLIT_MASTER_PATH = ROOT / "Streamlit_FE" / "data" / "provider_master_table.csv"


def _coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0.0)


def _robust_zscore(values: pd.Series) -> pd.Series:
    median = values.median()
    mad = (values - median).abs().median()
    if pd.isna(mad) or mad == 0:
        return pd.Series(0.0, index=values.index)
    return (values - median) / (1.4826 * mad)


def generate_provider_master_table() -> Path:
    fusion = pd.read_csv(OUTPUT_DIR / "final_provider_risk_scores.csv")
    iso = pd.read_csv(OUTPUT_DIR / "kaggle_iforest_scores.csv") if (OUTPUT_DIR / "kaggle_iforest_scores.csv").exists() else None
    ocsvm = pd.read_csv(OUTPUT_DIR / "kaggle_ocsvm_scores.csv") if (OUTPUT_DIR / "kaggle_ocsvm_scores.csv").exists() else None
    auto = pd.read_csv(OUTPUT_DIR / "autoencoder_provider_scores.csv") if (OUTPUT_DIR / "autoencoder_provider_scores.csv").exists() else None
    cat = pd.read_csv(ROOT / "catboost_model" / "output" / "provider_risk_ranked.csv") if (ROOT / "catboost_model" / "output" / "provider_risk_ranked.csv").exists() else None
    peer = pd.read_csv(ROOT / "peer-benchmarking" / "output" / "peer_benchmark_scores.csv") if (ROOT / "peer-benchmarking" / "output" / "peer_benchmark_scores.csv").exists() else None

    master = fusion[["Provider", "Final_Risk_Score", "Risk_Level", "Priority_Score"]].rename(
        columns={"Provider": "Provider_ID"}
    ).copy()

    if iso is not None:
        iso = iso[["Provider", "IForest_Risk"]].rename(columns={"Provider": "Provider_ID", "IForest_Risk": "IsolationForest_Score"})
        master = master.merge(iso, on="Provider_ID", how="left")
    else:
        master["IsolationForest_Score"] = 0.0

    if ocsvm is not None:
        ocsvm = ocsvm[["Provider", "OCSVM_Risk"]].rename(columns={"Provider": "Provider_ID", "OCSVM_Risk": "OCSVM_Score"})
        master = master.merge(ocsvm, on="Provider_ID", how="left")
    else:
        master["OCSVM_Score"] = 0.0

    if auto is not None:
        auto = auto[["Provider", "Autoencoder_Risk"]].rename(columns={"Provider": "Provider_ID", "Autoencoder_Risk": "Autoencoder_Score"})
        master = master.merge(auto, on="Provider_ID", how="left")
    else:
        master["Autoencoder_Score"] = 0.0

    if cat is not None:
        cat = cat[["Provider", "Risk_Score", "fraud_proba"]].rename(columns={"Provider": "Provider_ID", "Risk_Score": "CatBoost_Score"})
        master = master.merge(cat, on="Provider_ID", how="left")
    else:
        master["CatBoost_Score"] = 0.0

    if peer is not None:
        peer = peer[["Provider", "Peer_Risk_Score"]].rename(columns={"Provider": "Provider_ID", "Peer_Risk_Score": "Peer_Score"})
        master = master.merge(peer, on="Provider_ID", how="left")
    else:
        master["Peer_Score"] = 0.0

    features = pd.DataFrame()
    features_path = PROCESSED_DIR / "PROVIDER_ML_READY_KAGGLE.csv"
    if features_path.exists():
        features = pd.read_csv(features_path)
        if "Provider" in features.columns:
            features = features[["Provider", "Total_Reimbursement"]].rename(columns={"Provider": "Provider_ID", "Total_Reimbursement": "Total_Reimbursement"})
            master = master.merge(features, on="Provider_ID", how="left")

    if "Total_Reimbursement" not in master.columns:
        master["Total_Reimbursement"] = master["Final_Risk_Score"] * 1000.0

    master["Provider_Name"] = master["Provider_ID"]
    master["Provider_Type"] = "General Provider"
    master["IsolationForest_Score"] = _coerce_numeric(master.get("IsolationForest_Score", 0))
    master["Autoencoder_Score"] = _coerce_numeric(master.get("Autoencoder_Score", 0))
    master["Claim_Autoencoder_Score"] = _coerce_numeric(master.get("Final_Risk_Score", 0)) * 0.5
    master["CatBoost_Score"] = _coerce_numeric(master.get("CatBoost_Score", 0))
    master["Peer_Score"] = _coerce_numeric(master.get("Peer_Score", 0))
    master["Robust_ZScore"] = 0.0
    master["Percentile_Score"] = (master["Final_Risk_Score"].rank(pct=True, method="average") * 100).round(2)
    master["Potential_Leakage"] = (master["Total_Reimbursement"] * (master["Final_Risk_Score"] / 100.0)).round(2)

    if "Total_Reimbursement" in master.columns:
        robust = _robust_zscore(pd.to_numeric(master["Total_Reimbursement"], errors="coerce").fillna(0.0))
        master["Robust_ZScore"] = robust.round(4)

    master["Potential_Leakage"] = master["Potential_Leakage"].clip(lower=0)
    master["Priority_Score"] = _coerce_numeric(master.get("Priority_Score", 0.0)).clip(lower=0)
    master["Explanation"] = master.apply(
        lambda row: (
            f"Provider {row['Provider_ID']} scored {row['Final_Risk_Score']:.1f}/100 with risk level {row['Risk_Level']}. "
            f"Isolation Forest {row['IsolationForest_Score']:.1f}, Autoencoder {row['Autoencoder_Score']:.1f}, "
            f"CatBoost {row['CatBoost_Score']:.1f}, and peer risk {row['Peer_Score']:.1f}."
        ),
        axis=1,
    )

    required_columns = [
        "Provider_ID",
        "Provider_Name",
        "Provider_Type",
        "IsolationForest_Score",
        "Autoencoder_Score",
        "Claim_Autoencoder_Score",
        "CatBoost_Score",
        "Peer_Score",
        "Robust_ZScore",
        "Percentile_Score",
        "Potential_Leakage",
        "Final_Risk_Score",
        "Risk_Level",
        "Priority_Score",
        "Explanation",
    ]

    for col in required_columns:
        if col not in master.columns:
            master[col] = 0.0

    master = master[required_columns].sort_values("Final_Risk_Score", ascending=False).reset_index(drop=True)
    master.to_csv(MASTER_PATH, index=False)
    STREAMLIT_MASTER_PATH.parent.mkdir(parents=True, exist_ok=True)
    master.to_csv(STREAMLIT_MASTER_PATH, index=False)
    return MASTER_PATH


if __name__ == "__main__":
    output = generate_provider_master_table()
    print(f"Provider master generated: {output}")
