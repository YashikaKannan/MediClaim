#!/usr/bin/env python3
"""End-to-end explainable AI + RAG claim explanation pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from explanation_engine import run_explanation_engine
from rag_engine import build_rag_evidence


OUTPUT_DIR = "outputs/explanations"
CLAIM_EXPLANATIONS_FILE = "claim_explanations.csv"
DASHBOARD_FILE = "claim_explanations_dashboard.csv"


def setup_logging() -> None:
    """Configure logging for orchestration pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def classify_priority(risk_score: float) -> str:
    """Map risk score to queue priority label."""
    if risk_score >= 90:
        return "Critical"
    if risk_score >= 75:
        return "High"
    if risk_score >= 50:
        return "Medium"
    return "Low"


def generate_business_interpretation(row: pd.Series) -> str:
    """Generate concise business interpretation from rule + RAG evidence."""
    risk_score = float(row.get("RISK_SCORE", 0.0))
    risk_category = str(row.get("RISK_CATEGORY", "Unknown"))

    patterns: List[str] = []
    for i in range(1, 4):
        val = str(row.get(f"RETRIEVED_PATTERN_{i}", "")).strip()
        if val and val not in patterns:
            patterns.append(val)

    if not patterns:
        patterns = ["General anomaly profile"]

    pattern_text = ", ".join(patterns[:3])

    if risk_score >= 90:
        prefix = "This claim exhibits a concentrated high-risk anomaly signature"
    elif risk_score >= 75:
        prefix = "This claim shows substantial anomaly indicators"
    elif risk_score >= 50:
        prefix = "This claim has moderate anomaly characteristics"
    else:
        prefix = "This claim presents lower-severity anomaly indicators"

    return (
        f"{prefix} (risk={risk_score:.2f}, category={risk_category}) aligned with patterns such as "
        f"{pattern_text}. Prioritize clinical and billing documentation validation for payment integrity review."
    )


def build_final_claim_explanations(explanations_df: pd.DataFrame, rag_df: pd.DataFrame) -> pd.DataFrame:
    """Merge rule-based and RAG evidence into final explainable output."""
    merged = explanations_df.merge(rag_df, on="CLAIM_UID", how="left", suffixes=("", "_RAG"))

    if len(merged) != len(explanations_df):
        raise ValueError(
            "Final merge row-count mismatch detected. "
            f"Expected {len(explanations_df)}, got {len(merged)}."
        )

    tqdm.pandas(desc="Generating business interpretations")
    merged["BUSINESS_INTERPRETATION"] = merged.progress_apply(generate_business_interpretation, axis=1)

    final_cols = [
        "CLM_ID",
        "BENE_ID",
        "RISK_SCORE",
        "RISK_CATEGORY",
        "EXPLANATION_1",
        "EXPLANATION_2",
        "EXPLANATION_3",
        "BUSINESS_INTERPRETATION",
    ]

    for col in final_cols:
        if col not in merged.columns:
            merged[col] = ""

    final_df = merged[final_cols].copy()
    final_df = final_df.sort_values("RISK_SCORE", ascending=False)

    return final_df


def build_dashboard_output(final_df: pd.DataFrame) -> pd.DataFrame:
    """Create dashboard-ready output with summary text and triage priority."""
    dashboard_df = final_df.copy()
    dashboard_df["Priority"] = dashboard_df["RISK_SCORE"].apply(classify_priority)
    dashboard_df["Explanation Summary"] = (
        dashboard_df["EXPLANATION_1"].fillna("")
        + " | "
        + dashboard_df["EXPLANATION_2"].fillna("")
        + " | "
        + dashboard_df["EXPLANATION_3"].fillna("")
    )

    dashboard_df = dashboard_df.rename(
        columns={
            "RISK_SCORE": "Risk Score",
            "RISK_CATEGORY": "Risk Category",
        }
    )

    dashboard_cols = [
        "CLM_ID",
        "Risk Score",
        "Risk Category",
        "Explanation Summary",
        "Priority",
    ]

    return dashboard_df[dashboard_cols].sort_values("Risk Score", ascending=False)


def create_visualizations(
    explanations_df: pd.DataFrame,
    rag_df: pd.DataFrame,
    final_df: pd.DataFrame,
    output_dir: Path,
) -> Dict[str, Path]:
    """Create required explanation-layer visualizations."""
    output_dir.mkdir(parents=True, exist_ok=True)
    created_files: Dict[str, Path] = {}

    plt.style.use("seaborn-v0_8-whitegrid")

    # 1) Top explanation categories chart.
    plt.figure(figsize=(12, 6))
    cat_counts = explanations_df["TOP_EXPLANATION_CATEGORY"].value_counts().head(15)
    cat_counts.sort_values().plot(kind="barh", color="#2f6690")
    plt.title("Top Explanation Categories")
    plt.xlabel("Count")
    plt.ylabel("Explanation Category")
    plt.tight_layout()
    path1 = output_dir / "top_explanation_categories.png"
    plt.savefig(path1, dpi=300)
    plt.close()
    created_files["top_explanation_categories"] = path1

    # 2) Risk category distribution.
    plt.figure(figsize=(8, 5))
    risk_counts = final_df["RISK_CATEGORY"].value_counts().reindex(["Critical", "High", "Medium", "Low"], fill_value=0)
    risk_counts.plot(kind="bar", color=["#8b0000", "#d62828", "#f77f00", "#2a9d8f"])
    plt.title("Risk Category Distribution")
    plt.xlabel("Risk Category")
    plt.ylabel("Count")
    plt.tight_layout()
    path2 = output_dir / "risk_category_distribution_explanations.png"
    plt.savefig(path2, dpi=300)
    plt.close()
    created_files["risk_category_distribution"] = path2

    # 3) Top fraud pattern frequency chart from RAG retrieval.
    pattern_series = pd.concat(
        [
            rag_df["RETRIEVED_PATTERN_1"],
            rag_df["RETRIEVED_PATTERN_2"],
            rag_df["RETRIEVED_PATTERN_3"],
        ],
        ignore_index=True,
    ).dropna()
    pattern_series = pattern_series[pattern_series.astype(str).str.strip() != ""]

    plt.figure(figsize=(12, 6))
    pattern_counts = pattern_series.value_counts().head(15)
    pattern_counts.sort_values().plot(kind="barh", color="#6a994e")
    plt.title("Top Fraud Pattern Frequency")
    plt.xlabel("Count")
    plt.ylabel("Fraud Pattern")
    plt.tight_layout()
    path3 = output_dir / "top_fraud_pattern_frequency.png"
    plt.savefig(path3, dpi=300)
    plt.close()
    created_files["top_fraud_pattern_frequency"] = path3

    # 4) High-risk claims dashboard summary.
    dashboard_counts = final_df.assign(Priority=final_df["RISK_SCORE"].apply(classify_priority))["Priority"].value_counts()
    dashboard_counts = dashboard_counts.reindex(["Critical", "High", "Medium", "Low"], fill_value=0)

    top_claims = final_df[["CLM_ID", "RISK_SCORE"]].head(10).copy()
    top_claims["CLM_ID"] = top_claims["CLM_ID"].astype(str)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    axes[0].bar(dashboard_counts.index, dashboard_counts.values, color=["#7f0000", "#c1121f", "#fcbf49", "#2a9d8f"])
    axes[0].set_title("Claim Priority Breakdown")
    axes[0].set_xlabel("Priority")
    axes[0].set_ylabel("Claim Count")

    axes[1].barh(top_claims["CLM_ID"], top_claims["RISK_SCORE"], color="#003049")
    axes[1].invert_yaxis()
    axes[1].set_title("Top 10 Highest Risk Claims")
    axes[1].set_xlabel("Risk Score")
    axes[1].set_ylabel("CLM_ID")

    fig.suptitle("High-Risk Claims Dashboard Summary", fontsize=14)
    fig.tight_layout()
    path4 = output_dir / "high_risk_claims_dashboard_summary.png"
    fig.savefig(path4, dpi=300)
    plt.close(fig)
    created_files["high_risk_dashboard_summary"] = path4

    return created_files


def run_pipeline(base_dir: Path) -> None:
    """Execute complete explainability + RAG pipeline."""
    output_dir = base_dir / OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Running rule-based explanation engine.")
    explanations_df = run_explanation_engine(base_dir=base_dir)

    logging.info("Running FAISS RAG retrieval over fraud knowledge base.")
    rag_df = build_rag_evidence(claims_df=explanations_df, base_dir=base_dir)

    logging.info("Building final claim explanation outputs.")
    final_df = build_final_claim_explanations(explanations_df, rag_df)
    final_path = output_dir / CLAIM_EXPLANATIONS_FILE
    final_df.to_csv(final_path, index=False)

    dashboard_df = build_dashboard_output(final_df)
    dashboard_path = output_dir / DASHBOARD_FILE
    dashboard_df.to_csv(dashboard_path, index=False)

    viz_paths = create_visualizations(explanations_df, rag_df, final_df, output_dir)

    logging.info("Saved claim explanations: %s", final_path)
    logging.info("Saved dashboard-ready output: %s", dashboard_path)
    for name, path in viz_paths.items():
        logging.info("Saved plot [%s]: %s", name, path)


def main() -> None:
    """Program entrypoint."""
    setup_logging()
    try:
        base_dir = Path(__file__).resolve().parent
        run_pipeline(base_dir)
        logging.info("Explainable AI + RAG pipeline completed successfully.")
    except Exception as exc:
        logging.exception("Claim explanation generation failed: %s", exc)
        raise


if __name__ == "__main__":
    main()
