"""
graph_viz.py — draws the fraud-ring picture (providers sharing patients).

Saves output/fraud_ring_graph.png. Called automatically at the end of train.py,
or run on its own:  python -m src.graph_viz
"""
from pathlib import Path
import warnings
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .features import build_features

warnings.filterwarnings("ignore")
DATA = Path(__file__).resolve().parents[1] / "data"
OUTPUT = Path(__file__).resolve().parents[1] / "output"
OUTPUT.mkdir(exist_ok=True)


def draw(top_n_providers: int = 6, max_patients: int = 25):
    prov = build_features()

    inp = pd.read_csv(DATA / "train_inpatient.csv", low_memory=False)
    out = pd.read_csv(DATA / "train_outpatient.csv", low_memory=False)
    inp["is_inpatient"] = 1
    out["is_inpatient"] = 0
    common = list((set(inp.columns) & set(out.columns)) - {"is_inpatient"})
    claims = pd.concat([inp[common + ["is_inpatient"]], out[common + ["is_inpatient"]]],
                       ignore_index=True)

    # most-connected fraud providers -> the clearest ring
    fraud = (prov[prov["y"] == 1]
             .sort_values("ring_connections", ascending=False)
             .head(top_n_providers)["Provider"].tolist())
    sub = claims[claims["Provider"].isin(fraud)][["Provider", "BeneID"]].drop_duplicates()
    shared = sub.groupby("BeneID")["Provider"].nunique()
    shared = shared[shared >= 2].index[:max_patients]     # patients shared by >=2 providers
    sub = sub[sub["BeneID"].isin(shared)]

    if sub.empty:
        print("[graph] no shared-patient ring found to draw.")
        return

    G = nx.Graph()
    for _, r in sub.iterrows():
        G.add_node(r["Provider"], kind="provider")
        G.add_node(r["BeneID"], kind="patient")
        G.add_edge(r["Provider"], r["BeneID"])
    provs = [n for n, d in G.nodes(data=True) if d["kind"] == "provider"]
    pats = [n for n, d in G.nodes(data=True) if d["kind"] == "patient"]

    plt.figure(figsize=(13, 9))
    pos = nx.spring_layout(G, k=0.5, seed=42)
    nx.draw_networkx_edges(G, pos, alpha=0.3, edge_color="gray")
    nx.draw_networkx_nodes(G, pos, nodelist=provs, node_color="#d62728",
                           node_size=1400, node_shape="s", label="Provider (fraud)")
    nx.draw_networkx_nodes(G, pos, nodelist=pats, node_color="#1f77b4",
                           node_size=350, label="Patient")
    nx.draw_networkx_labels(G, pos, labels={n: n for n in provs},
                            font_size=7, font_color="white", font_weight="bold")
    plt.title("Fraud-Ring Graph: providers (red) sharing the same patients (blue)\n"
              "Shared patients = suspicious coordinated billing", fontsize=13)
    plt.legend(scatterpoints=1, fontsize=11)
    plt.axis("off")
    plt.tight_layout()
    out_path = OUTPUT / "fraud_ring_graph.png"
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"[graph] saved -> {out_path}")


if __name__ == "__main__":
    draw()
