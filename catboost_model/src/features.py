"""
features.py — build the provider feature table, including the two "wow" features:
  * ring_connections  (fraud-ring graph: how many providers share >=5 patients)
  * max_month_jump    (temporal drift: biggest month-over-month claim-volume jump)

All columns from the raw files are kept; we only READ the ones we need.
"""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import networkx as nx

warnings.filterwarnings("ignore")
DATA = Path(__file__).resolve().parents[1] / "data"

FEATURES = [
    "n_claims", "tot_reimb", "avg_reimb", "std_reimb", "n_benes", "n_phys",
    "pct_inpatient", "n_after_death", "avg_chronic", "avg_ndiag", "uniq_diag",
    "claims_per_bene", "max_month_jump", "ring_connections",
]


def _load():
    ben = pd.read_csv(DATA / "train_beneficiary.csv", low_memory=False)
    inp = pd.read_csv(DATA / "train_inpatient.csv", low_memory=False)
    out = pd.read_csv(DATA / "train_outpatient.csv", low_memory=False)
    lab = pd.read_csv(DATA / "train_labels.csv", low_memory=False)
    return ben, inp, out, lab


def build_features() -> pd.DataFrame:
    ben, inp, out, lab = _load()

    inp["is_inpatient"] = 1
    out["is_inpatient"] = 0
    common = list((set(inp.columns) & set(out.columns)) - {"is_inpatient"})
    claims = pd.concat([inp[common + ["is_inpatient"]], out[common + ["is_inpatient"]]],
                       ignore_index=True)

    chron = [c for c in ben.columns if c.startswith("ChronicCond")]
    for c in chron:
        ben[c] = (ben[c] == 1).astype(int)
    ben["DOD"] = pd.to_datetime(ben["DOD"], errors="coerce")
    ben["ChronicCount"] = ben[chron].sum(axis=1)

    claims["ClaimStartDt"] = pd.to_datetime(claims["ClaimStartDt"], errors="coerce")
    claims = claims.merge(ben[["BeneID", "DOD", "ChronicCount"]], on="BeneID", how="left")
    claims["claim_after_death"] = (
        (claims["DOD"].notna()) & (claims["ClaimStartDt"] > claims["DOD"])
    ).astype(int)
    diag = [c for c in claims.columns if c.startswith("ClmDiagnosisCode_")]
    claims["n_diag"] = claims[diag].notna().sum(axis=1)
    claims["month"] = claims["ClaimStartDt"].dt.to_period("M").astype(str)

    g = claims.groupby("Provider")
    prov = pd.DataFrame({
        "n_claims": g.size(),
        "tot_reimb": g["InscClaimAmtReimbursed"].sum(),
        "avg_reimb": g["InscClaimAmtReimbursed"].mean(),
        "std_reimb": g["InscClaimAmtReimbursed"].std(),
        "n_benes": g["BeneID"].nunique(),
        "n_phys": g["AttendingPhysician"].nunique(),
        "pct_inpatient": g["is_inpatient"].mean(),
        "n_after_death": g["claim_after_death"].sum(),
        "avg_chronic": g["ChronicCount"].mean(),
        "avg_ndiag": g["n_diag"].mean(),
        "uniq_diag": g["ClmDiagnosisCode_1"].nunique(),
    }).reset_index()
    prov["claims_per_bene"] = prov["n_claims"] / prov["n_benes"]

    # ---- TEMPORAL DRIFT: biggest month-over-month jump in claim volume ----
    mv = claims.groupby(["Provider", "month"]).size().reset_index(name="c")
    def max_jump(s):
        v = s.sort_values("month")["c"].values
        return int(np.max(np.diff(v))) if len(v) > 1 else 0
    drift = mv.groupby("Provider").apply(max_jump).reset_index(name="max_month_jump")
    prov = prov.merge(drift, on="Provider", how="left")

    # ---- FRAUD-RING GRAPH: providers linked by >=5 shared patients ----
    pb = claims[["Provider", "BeneID"]].drop_duplicates()
    sh = pb.merge(pb, on="BeneID")
    sh = sh[sh["Provider_x"] < sh["Provider_y"]]
    edge = sh.groupby(["Provider_x", "Provider_y"]).size().reset_index(name="w")
    edge = edge[edge["w"] >= 5]
    G = nx.from_pandas_edgelist(edge, "Provider_x", "Provider_y", "w")
    deg = dict(G.degree())
    prov["ring_connections"] = prov["Provider"].map(deg).fillna(0)

    prov = prov.fillna(0)
    prov = prov.merge(lab, on="Provider", how="inner")
    prov["y"] = (prov["PotentialFraud"] == "Yes").astype(int)
    return prov
