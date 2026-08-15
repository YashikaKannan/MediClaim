"""
lof_provider_anomaly.py — Member 5 (LOF provider anomaly detection).

Reads the ready-made PROVIDER_ML_READY.csv (1 row = 1 provider) and runs the
full spec: per-specialty LOF -> 0-100 risk score -> risk bands -> explainable
peer-deviation reasons -> LEIE validation -> member5_lof_results.csv.

Run:  python -m src.lof_provider_anomaly
"""
import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler

from . import config as C


def _risk_level(score: float) -> str:
    for upper, label in C.RISK_BANDS:
        if score <= upper:
            return label
    return C.RISK_BANDS[-1][1]


def _reasons(row: pd.Series, med: pd.Series) -> str:
    bits = []
    pay = row["Tot_Mdcr_Pymt_Amt"] / med["Tot_Mdcr_Pymt_Amt"] if med["Tot_Mdcr_Pymt_Amt"] else np.nan
    svc = row["Services_Per_Beneficiary"] / med["Services_Per_Beneficiary"] if med["Services_Per_Beneficiary"] else np.nan
    chg = row["Charge_Per_Service"] / med["Charge_Per_Service"] if med["Charge_Per_Service"] else np.nan
    if pay and pay >= 1.5:
        bits.append(f"Medicare payment {pay:.1f}x specialty peer median")
    if svc and svc >= 1.5:
        bits.append(f"Services per beneficiary {svc:.1f}x peer median")
    if chg and chg >= 1.5:
        bits.append(f"Charge per service {chg:.1f}x peer median")
    if row["Payment_to_Allowed_Ratio"] >= med["Payment_to_Allowed_Ratio"] * 1.15:
        bits.append("Payment-to-allowed ratio exceeds peer median")
    return "; ".join(bits) if bits else "Multivariate pattern differs from specialty peers"


def run() -> pd.DataFrame:
    df = pd.read_csv(C.PROVIDER_ML_READY_CSV, low_memory=False)

    feat_t = [c + "_t" if c in C.LOG_COLS else c for c in C.FEATURES]
    scored_parts, review_parts = [], []

    for spec, g in df.groupby("Rndrng_Prvdr_Type"):
        g = g.copy()

        if len(g) < C.MIN_PEERS:                                   # STEP 3
            g["LOF_Score"] = np.nan
            g["Risk_Score"] = np.nan
            g["Risk_Level"] = "INSUFFICIENT_PEERS"
            g["Explanation"] = "Specialty has fewer than MIN_PEERS providers; manual review"
            review_parts.append(g)
            continue

        g[C.FEATURES] = g[C.FEATURES].fillna(g[C.FEATURES].median())  # STEP 4
        for c in C.LOG_COLS:                                          # STEP 5
            g[c + "_t"] = np.log1p(g[c])
        X = RobustScaler().fit_transform(g[feat_t])                   # STEP 6

        n = min(C.N_NEIGHBORS, len(g) - 1)                           # STEP 7
        lof = LocalOutlierFactor(n_neighbors=n, contamination=C.CONTAMINATION)
        lof.fit_predict(X)
        g["LOF_Score"] = -lof.negative_outlier_factor_               # STEP 8

        med = g[C.FEATURES].median()                                 # STEP 10
        g["Payment_Ratio"]     = g["Tot_Mdcr_Pymt_Amt"] / med["Tot_Mdcr_Pymt_Amt"]
        g["Service_Ratio"]     = g["Services_Per_Beneficiary"] / med["Services_Per_Beneficiary"]
        g["Beneficiary_Ratio"] = g["Tot_Benes"] / med["Tot_Benes"]
        g["Explanation"]       = g.apply(lambda r: _reasons(r, med), axis=1)
        scored_parts.append(g)

    scored = pd.concat(scored_parts, ignore_index=True)

    lo = scored["LOF_Score"].min()                                   # STEP 9
    cap = scored["LOF_Score"].quantile(C.SCORE_CAP_Q)
    scored["Risk_Score"] = ((scored["LOF_Score"].clip(upper=cap) - lo) / (cap - lo) * 100).round(2)
    scored["Risk_Level"] = scored["Risk_Score"].apply(_risk_level)

    result = pd.concat([scored] + review_parts, ignore_index=True)
    result = _leie_overlap(result)

    cols = ["Rndrng_NPI", "Rndrng_Prvdr_Type", "LOF_Score", "Risk_Score", "Risk_Level",
            "Payment_Ratio", "Service_Ratio", "Beneficiary_Ratio",
            "LEIE_Match", "Currently_Excluded", "Fraud_Related", "Explanation"]
    result = result.reindex(columns=cols).sort_values("Risk_Score", ascending=False, na_position="last")
    result.to_csv(C.RESULTS_CSV, index=False)

    _report(result)
    return result


def _leie_overlap(result: pd.DataFrame) -> pd.DataFrame:
    le = pd.read_csv(C.LEIE_CSV, low_memory=False)
    le["_npi"] = pd.to_numeric(le["NPI"], errors="coerce")
    le = le.dropna(subset=["_npi"])
    le["_npi"] = le["_npi"].astype("int64").astype(str)

    excluded = set(le["_npi"])
    curr = set(le.loc[le.get("IS_CURRENTLY_EXCLUDED", False) == True, "_npi"]) if "IS_CURRENTLY_EXCLUDED" in le else set()
    fraud = set(le.loc[le.get("IS_FRAUD_RELATED_EXCLUSION", False) == True, "_npi"]) if "IS_FRAUD_RELATED_EXCLUSION" in le else set()

    npi = result["Rndrng_NPI"].astype("int64").astype(str)
    result["LEIE_Match"]        = npi.isin(excluded)
    result["Currently_Excluded"] = npi.isin(curr)
    result["Fraud_Related"]      = npi.isin(fraud)
    return result


def _report(result: pd.DataFrame) -> None:
    scored = result[result["Risk_Level"] != "INSUFFICIENT_PEERS"]
    print("\n================ VALIDATION (unsupervised) ================")
    print(f"Total providers analyzed : {len(result):,}")
    print(f"Scored                   : {len(scored):,}")
    print(f"Insufficient peers       : {(result['Risk_Level']=='INSUFFICIENT_PEERS').sum():,}")
    print("Risk levels:")
    print(scored["Risk_Level"].value_counts().reindex(["Low", "Medium", "High", "Critical"]).to_string())
    for q, lbl in [(0.99, "Top 1%"), (0.97, "Top 3%"), (0.95, "Top 5%")]:
        sub = scored[scored["Risk_Score"] >= scored["Risk_Score"].quantile(q)]
        print(f"{lbl:>7}: {len(sub):>5} providers | LEIE {int(sub['LEIE_Match'].sum())} | fraud-related {int(sub['Fraud_Related'].sum())}")
    print(f"Total LEIE overlap       : {int(scored['LEIE_Match'].sum())}")
    print(f"Output written           : {C.RESULTS_CSV}")
    print("==========================================================")


if __name__ == "__main__":
    run()
