#!/usr/bin/env python3
"""Rule-based explainability engine for suspicious healthcare claims."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


DATASET_FILE = "carrier_clean_final_perfect.csv"
SUSPICIOUS_FILE = "outputs/suspicious_claims_3pct.csv"
OUTPUT_DIR = "outputs/explanations"
INTERMEDIATE_FILE = "suspicious_claims_with_metrics.csv"
THRESHOLDS_FILE = "feature_thresholds.json"

KEY_COLUMNS = ["CLM_ID", "BENE_ID", "PRVDR_SPCLTY"]

NUMERIC_COLUMNS = [
    "CLM_PMT_AMT",
    "NCH_CARR_CLM_SBMTD_CHRG_AMT",
    "NCH_CARR_CLM_ALOWD_AMT",
    "LINE_SBMTD_CHRG_AMT",
    "LINE_ALOWD_CHRG_AMT",
    "LINE_SRVC_CNT",
    "DIAGNOSIS_COUNT",
    "LINE_NCH_PMT_AMT",
]

FEATURE_COLUMNS = [
    "CLM_PMT_AMT",
    "NCH_CARR_CLM_SBMTD_CHRG_AMT",
    "NCH_CARR_CLM_ALOWD_AMT",
    "LINE_SBMTD_CHRG_AMT",
    "LINE_ALOWD_CHRG_AMT",
    "LINE_SRVC_CNT",
    "DIAGNOSIS_COUNT",
    "CHARGE_ALLOWED_RATIO",
    "CHARGE_PAYMENT_RATIO",
    "SERVICE_INTENSITY",
    "PEER_BILLING_RATIO",
    "EXCESS_BILLING_RATIO",
]


@dataclass
class RuleDefinition:
    """Defines a single explanation rule and how to evaluate it."""

    code: str
    feature: str
    threshold_key: str
    operator: str
    category: str
    explanation_template: str
    business_signal: str
    weight: float


RULES: List[RuleDefinition] = [
    RuleDefinition(
        code="R1",
        feature="CLM_PMT_AMT",
        threshold_key="p99",
        operator=">",
        category="Excessive Claim Payment",
        explanation_template="Claim payment amount is in the top 1% of all claims (value={value:.2f}, p99={threshold:.2f}).",
        business_signal="High payout outlier",
        weight=1.0,
    ),
    RuleDefinition(
        code="R2",
        feature="NCH_CARR_CLM_SBMTD_CHRG_AMT",
        threshold_key="p99",
        operator=">",
        category="Excessive Submitted Charges",
        explanation_template="Submitted claim charges are among the highest observed (value={value:.2f}, p99={threshold:.2f}).",
        business_signal="Charge inflation risk",
        weight=0.95,
    ),
    RuleDefinition(
        code="R3",
        feature="CHARGE_ALLOWED_RATIO",
        threshold_key="p95",
        operator=">",
        category="High Charge Allowed Ratio",
        explanation_template="Submitted charges are significantly higher than allowed charges (ratio={value:.2f}, p95={threshold:.2f}).",
        business_signal="Potential overbilling",
        weight=0.9,
    ),
    RuleDefinition(
        code="R4",
        feature="CHARGE_PAYMENT_RATIO",
        threshold_key="p95",
        operator=">",
        category="High Charge Payment Ratio",
        explanation_template="Submitted charges are disproportionately high compared with paid amounts (ratio={value:.2f}, p95={threshold:.2f}).",
        business_signal="High denied or downcoded billing pressure",
        weight=0.88,
    ),
    RuleDefinition(
        code="R5",
        feature="SERVICE_INTENSITY",
        threshold_key="p95",
        operator=">",
        category="Unusual Service Intensity",
        explanation_template="Service volume is unusually high relative to diagnosis complexity (value={value:.2f}, p95={threshold:.2f}).",
        business_signal="Utilization anomaly",
        weight=0.84,
    ),
    RuleDefinition(
        code="R6",
        feature="DIAGNOSIS_COUNT",
        threshold_key="p99",
        operator=">",
        category="Abnormally High Diagnosis Count",
        explanation_template="Diagnosis count is in the extreme tail (value={value:.0f}, p99={threshold:.0f}).",
        business_signal="Possible coding intensity inflation",
        weight=0.8,
    ),
    RuleDefinition(
        code="R7",
        feature="EXCESS_BILLING_RATIO",
        threshold_key="p95",
        operator=">",
        category="Potential Overbilling",
        explanation_template="Claim-level submitted charges materially exceed allowed amounts (ratio={value:.2f}, p95={threshold:.2f}).",
        business_signal="Claim-level billing excess",
        weight=0.9,
    ),
    RuleDefinition(
        code="R8",
        feature="PEER_BILLING_RATIO",
        threshold_key="p95",
        operator=">",
        category="Potential Upcoding",
        explanation_template="Billing is significantly above specialty peers (peer ratio={value:.2f}, p95={threshold:.2f}).",
        business_signal="Specialty peer deviation",
        weight=0.86,
    ),
    RuleDefinition(
        code="R9",
        feature="LINE_SRVC_CNT",
        threshold_key="p99",
        operator=">",
        category="Excessive Service Utilization",
        explanation_template="Service count is within the top 1% of claims (value={value:.0f}, p99={threshold:.0f}).",
        business_signal="High service utilization",
        weight=0.82,
    ),
    RuleDefinition(
        code="R10",
        feature="COMPOSITE_FINANCIAL_OUTLIER",
        threshold_key="fixed",
        operator=">=",
        category="Extreme Financial Outlier",
        explanation_template="Multiple financial dimensions are simultaneously extreme (outlier count={value:.0f}).",
        business_signal="Multi-feature financial anomaly",
        weight=1.1,
    ),
]


def setup_logging() -> None:
    """Configure logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Safely divide two vectors and handle infinities and missing values."""
    result = numerator.astype("float64") / denominator.astype("float64")
    return result.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def load_inputs(base_dir: Path, dataset_file: str, suspicious_file: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load dataset and suspicious claims outputs."""
    dataset_path = base_dir / dataset_file
    suspicious_path = base_dir / suspicious_file

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    if not suspicious_path.exists():
        raise FileNotFoundError(f"Suspicious claims file not found: {suspicious_path}")

    logging.info("Loading full claim dataset: %s", dataset_path)
    df = pd.read_csv(dataset_path, low_memory=False)

    logging.info("Loading suspicious claims: %s", suspicious_path)
    suspicious_df = pd.read_csv(suspicious_path, low_memory=False)

    return df, suspicious_df


def validate_columns(df: pd.DataFrame, suspicious_df: pd.DataFrame) -> None:
    """Validate all required columns for explanation processing."""
    required_dataset_cols = set(KEY_COLUMNS + NUMERIC_COLUMNS)
    missing_dataset = sorted([c for c in required_dataset_cols if c not in df.columns])
    if missing_dataset:
        raise ValueError(f"Missing required columns in dataset: {missing_dataset}")

    required_suspicious_cols = {"CLM_ID", "BENE_ID", "RISK_SCORE", "RISK_CATEGORY", "ANOMALY_FLAG"}
    missing_suspicious = sorted([c for c in required_suspicious_cols if c not in suspicious_df.columns])
    if missing_suspicious:
        raise ValueError(f"Missing required columns in suspicious claims file: {missing_suspicious}")


def clean_and_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Clean core fields and compute engineered anomaly features."""
    logging.info("Cleaning numeric columns and engineering features.")

    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())

    df["PRVDR_SPCLTY"] = df["PRVDR_SPCLTY"].fillna("UNKNOWN")

    # Detects line-level submitted charge excess over allowed reimbursement.
    df["CHARGE_ALLOWED_RATIO"] = safe_divide(
        df["LINE_SBMTD_CHRG_AMT"], df["LINE_ALOWD_CHRG_AMT"] + 1.0
    )

    # Captures mismatch between submitted charges and paid amounts.
    df["CHARGE_PAYMENT_RATIO"] = safe_divide(
        df["LINE_SBMTD_CHRG_AMT"], df["LINE_NCH_PMT_AMT"] + 1.0
    )

    # Measures service count intensity relative to diagnosis burden.
    df["SERVICE_INTENSITY"] = safe_divide(df["LINE_SRVC_CNT"], df["DIAGNOSIS_COUNT"] + 1.0)

    # Measures claim-level submitted-vs-allowed billing spread.
    df["EXCESS_BILLING_RATIO"] = safe_divide(
        df["NCH_CARR_CLM_SBMTD_CHRG_AMT"], df["NCH_CARR_CLM_ALOWD_AMT"] + 1.0
    )

    specialty_median = df.groupby("PRVDR_SPCLTY")["LINE_SBMTD_CHRG_AMT"].transform("median")
    specialty_median = specialty_median.replace(0, np.nan)

    # Benchmarks provider billing against specialty-level typical submitted amounts.
    df["PEER_BILLING_RATIO"] = safe_divide(df["LINE_SBMTD_CHRG_AMT"], specialty_median)

    return df


def compute_thresholds(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Compute median and tail-percentile thresholds for explanation rules."""
    logging.info("Computing median and percentile thresholds.")

    thresholds: Dict[str, Dict[str, float]] = {}
    quantiles = [0.5, 0.9, 0.95, 0.99]

    for feature in FEATURE_COLUMNS:
        values = pd.to_numeric(df[feature], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if values.empty:
            thresholds[feature] = {"median": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0}
            continue

        qvals = values.quantile(quantiles)
        thresholds[feature] = {
            "median": float(qvals.loc[0.5]),
            "p90": float(qvals.loc[0.9]),
            "p95": float(qvals.loc[0.95]),
            "p99": float(qvals.loc[0.99]),
        }

    return thresholds


def create_composite_outlier_flag(df: pd.DataFrame, thresholds: Dict[str, Dict[str, float]]) -> pd.Series:
    """Count how many key financial metrics cross 99th percentile simultaneously."""
    high_cols = [
        "CLM_PMT_AMT",
        "NCH_CARR_CLM_SBMTD_CHRG_AMT",
        "LINE_SBMTD_CHRG_AMT",
        "EXCESS_BILLING_RATIO",
    ]

    outlier_count = pd.Series(0, index=df.index, dtype="int64")
    for col in high_cols:
        p99 = thresholds[col]["p99"]
        outlier_count += (df[col] > p99).astype("int64")

    return outlier_count


def prepare_suspicious_frame(df: pd.DataFrame, suspicious_df: pd.DataFrame) -> pd.DataFrame:
    """Join suspicious claims back to original records for feature-level explanations."""
    logging.info("Joining suspicious claims with original records.")

    merged_df = suspicious_df[suspicious_df["ANOMALY_FLAG"] == -1].copy()
    merged_df = merged_df.reset_index(drop=True)
    merged_df["CLAIM_UID"] = merged_df.index.astype("int64")
    merged_df["CLM_ID"] = merged_df["CLM_ID"].astype(str)
    merged_df["BENE_ID"] = merged_df["BENE_ID"].astype(str)

    source_df = df.copy()
    source_df["CLM_ID"] = source_df["CLM_ID"].astype(str)
    source_df["BENE_ID"] = source_df["BENE_ID"].astype(str)

    keep_cols = list(set(KEY_COLUMNS + FEATURE_COLUMNS))
    keep_cols = [c for c in keep_cols if c in source_df.columns]

    source_unique = source_df[keep_cols].drop_duplicates(subset=["CLM_ID", "BENE_ID"], keep="first")

    result = merged_df.merge(source_unique, on=["CLM_ID", "BENE_ID"], how="left", suffixes=("", "_ORIG"))

    # Prefer specialty from source if available.
    if "PRVDR_SPCLTY_ORIG" in result.columns:
        result["PRVDR_SPCLTY"] = result["PRVDR_SPCLTY_ORIG"].fillna(result.get("PRVDR_SPCLTY", "UNKNOWN"))
        result = result.drop(columns=["PRVDR_SPCLTY_ORIG"])

    return result


def evaluate_rule(value: float, threshold: float, operator: str) -> bool:
    """Evaluate a rule condition based on operator semantics."""
    if operator == ">":
        return value > threshold
    if operator == ">=":
        return value >= threshold
    if operator == "<":
        return value < threshold
    if operator == "<=":
        return value <= threshold
    raise ValueError(f"Unsupported operator: {operator}")


def build_claim_explanations(
    suspicious_features_df: pd.DataFrame,
    thresholds: Dict[str, Dict[str, float]],
) -> pd.DataFrame:
    """Generate top-3 claim explanations from percentile-based anomaly rules."""
    logging.info("Generating claim-level explanations with percentile rules.")

    rows: List[Dict[str, object]] = []

    tqdm.pandas(desc="Scoring explanation rules")

    composite_col = "COMPOSITE_FINANCIAL_OUTLIER"
    suspicious_features_df[composite_col] = create_composite_outlier_flag(suspicious_features_df, thresholds)

    for row in tqdm(
        suspicious_features_df.itertuples(index=False),
        total=len(suspicious_features_df),
        desc="Building explanations",
    ):
        row_dict = row._asdict()
        triggered: List[Tuple[float, str, str]] = []

        for rule in RULES:
            if rule.threshold_key == "fixed":
                threshold_val = 2.0
            else:
                threshold_val = thresholds[rule.feature][rule.threshold_key]

            value = float(row_dict.get(rule.feature, 0.0) or 0.0)
            if evaluate_rule(value, threshold_val, rule.operator):
                # Scores prioritize both rule weight and distance above threshold.
                normalized_gap = (value - threshold_val) / (abs(threshold_val) + 1e-6)
                score = rule.weight * (1.0 + max(0.0, normalized_gap))

                explanation = rule.explanation_template.format(value=value, threshold=threshold_val)
                triggered.append((score, explanation, rule.category))

        triggered.sort(key=lambda x: x[0], reverse=True)

        if not triggered:
            triggered = [
                (
                    0.1,
                    "Combined anomaly profile differs from normal claim patterns based on financial and utilization features.",
                    "General Anomaly Signal",
                )
            ]

        top_three = triggered[:3]
        explanations = [item[1] for item in top_three]
        categories = [item[2] for item in top_three]

        while len(explanations) < 3:
            explanations.append("Additional supporting anomaly signals detected in claim profile.")
            categories.append("Supporting Signal")

        anomaly_summary = " | ".join(explanations)

        rows.append(
            {
                "CLAIM_UID": int(row_dict.get("CLAIM_UID", -1)),
                "CLM_ID": row_dict.get("CLM_ID"),
                "BENE_ID": row_dict.get("BENE_ID"),
                "PRVDR_SPCLTY": row_dict.get("PRVDR_SPCLTY", "UNKNOWN"),
                "RISK_SCORE": float(row_dict.get("RISK_SCORE", 0.0)),
                "RISK_CATEGORY": row_dict.get("RISK_CATEGORY", "Unknown"),
                "ANOMALY_FLAG": int(row_dict.get("ANOMALY_FLAG", -1)),
                "EXPLANATION_1": explanations[0],
                "EXPLANATION_2": explanations[1],
                "EXPLANATION_3": explanations[2],
                "TOP_EXPLANATION_CATEGORY": categories[0],
                "ANOMALY_SUMMARY": anomaly_summary,
                "TRIGGERED_RULE_CATEGORIES": json.dumps(categories),
            }
        )

    return pd.DataFrame(rows)


def save_thresholds(thresholds: Dict[str, Dict[str, float]], output_path: Path) -> None:
    """Persist threshold metadata for traceability and governance."""
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(thresholds, f, indent=2)


def run_explanation_engine(
    base_dir: Path,
    dataset_file: str = DATASET_FILE,
    suspicious_file: str = SUSPICIOUS_FILE,
    output_dir: str = OUTPUT_DIR,
) -> pd.DataFrame:
    """Execute full explanation pipeline and save intermediate outputs."""
    output_path = base_dir / output_dir
    output_path.mkdir(parents=True, exist_ok=True)

    df, suspicious_df = load_inputs(base_dir, dataset_file, suspicious_file)
    validate_columns(df, suspicious_df)

    df = clean_and_engineer(df)
    thresholds = compute_thresholds(df)

    suspicious_features_df = prepare_suspicious_frame(df, suspicious_df)
    explanations_df = build_claim_explanations(suspicious_features_df, thresholds)

    explanations_df.to_csv(output_path / INTERMEDIATE_FILE, index=False)
    save_thresholds(thresholds, output_path / THRESHOLDS_FILE)

    logging.info("Saved intermediate explained suspicious claims to: %s", output_path / INTERMEDIATE_FILE)
    logging.info("Saved feature thresholds to: %s", output_path / THRESHOLDS_FILE)

    return explanations_df


def main() -> None:
    """CLI entrypoint."""
    setup_logging()
    try:
        base_dir = Path(__file__).resolve().parent
        _ = run_explanation_engine(base_dir)
        logging.info("Explanation engine completed successfully.")
    except Exception as exc:
        logging.exception("Explanation engine failed: %s", exc)
        raise


if __name__ == "__main__":
    main()
